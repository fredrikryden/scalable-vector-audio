// Concern: proves an argument reaches its slot as the number its text names | Non-concern: what a builtin does with the number (vocabulary.rs) | IO: (a composition) -> the number that arrived

use crate::fixtures::graph_of;
use sva_engine::{EngineError, RenderConfig, Tier, render, types};

/// A time constant of zero, as a glide of no length is, divides the instant by zero, and a q
/// of zero divides a filter's damping by it: each is refused in writing, never a panic.
#[test]
fn a_rate_or_a_damping_divided_by_zero_refuses() {
    for (written, code) in [
        (
            "gt = 0\ncrop(sin(2*pi*300*t)*exp(-t/gt), 0s, 1s)\n",
            "collapse.not_evaluable",
        ),
        ("crop(sin(t/0), 0s, 1s)\n", "collapse.not_evaluable"),
        (
            "crop(exp(-t*t*1e200*1e200), 0s, 1s)\n",
            "collapse.not_evaluable",
        ),
        (
            "crop(lowpass(sin(2*pi*300*t), cutoff=500, q=0), 0s, 1s)\n",
            "engine.filter_without_response",
        ),
    ] {
        let g = graph_of("no-rate", &[("body", written)]);
        let Err(refused) = render(
            &g,
            "body",
            RenderConfig::seconds(44_100, 0.01),
            &Tier::default(),
        ) else {
            panic!("{written} names no number and renders nothing");
        };
        assert_eq!(refused.code(), code, "{written}: {refused}");
    }
}

/// FORMAT 15.3: a ref naming one number is that number wherever it is written. A window's
/// bounds are two numbers, so a constant expression folds into one; only a bound that moves
/// with the free variable names no window.
#[test]
fn a_crop_bound_may_be_a_constant_expression() {
    let g = graph_of(
        "crop-bound",
        &[
            ("variables/bar", "0.5\n"),
            ("tone", "sin(2*pi*440*t)\n"),
            ("written", "crop(@tone(t), 0s, 1.6s - 0.1s)\n"),
            ("read", "crop(@tone(t), 0s, @variables/bar*3)\n"),
        ],
    );
    let ends = |node: &str| {
        let held = render(
            &g,
            node,
            RenderConfig::seconds(8_192, 2.0),
            &Tier::default(),
        )
        .unwrap_or_else(|e| panic!("{node}: {e}"));
        let id = held.id(node).unwrap_or_else(|| panic!("{node} typed"));
        let plane = held.output(id).expect("a rendered law").plane(0).to_vec();
        let last = plane.iter().rposition(|s| *s != 0.0).expect("some sound");
        (last + 1) as f64 / 8_192.0
    };
    assert!(
        (ends("written") - 1.5).abs() < 1e-3,
        "1.6s - 0.1s is 1.5s, got {}",
        ends("written")
    );
    assert!(
        (ends("read") - 1.5).abs() < 1e-3,
        "a bar of 0.5s three times over is 1.5s, got {}",
        ends("read")
    );
}

fn plane_of(g: &sva_ast::Graph, node: &str) -> Vec<f64> {
    let held = render(
        g,
        node,
        RenderConfig::seconds(44_100, 0.02),
        &Tier::default(),
    )
    .unwrap_or_else(|e| panic!("{node}: {e}"));
    let id = held.id(node).unwrap_or_else(|| panic!("{node} typed"));
    held.output(id).expect("a rendered solve").plane(0).to_vec()
}

/// The bug this closes: a negative whole exponent of a constant lowered to a pole atom, which
/// no fold reads as a number, so `b=` fell back to its default for every such exponent.
#[test]
fn a_negative_whole_power_inside_a_solver_argument_is_the_number_it_names() {
    let x = 261.63_f64 / 262.0;
    let inv = |k: f64| 1.0 / k;
    for (exponent, reciprocal) in [
        ("-1", inv(x)),
        ("-1.0", inv(x)),
        ("-2", inv(x * x)),
        ("-3", inv(x * x * x)),
    ] {
        let g = graph_of(
            "negative-power",
            &[
                (
                    "powered",
                    &format!(
                        "f0 = 261.63\nchaigne_askenfelt(f0, b=0.0001*pow(f0/262, {exponent}))\n"
                    ),
                ),
                (
                    "spelled",
                    &format!("chaigne_askenfelt(261.63, b=0.0001*{reciprocal:?})\n"),
                ),
                ("omitted", "chaigne_askenfelt(261.63)\n"),
            ],
        );
        let powered = plane_of(&g, "powered");
        assert_eq!(
            powered,
            plane_of(&g, "spelled"),
            "pow(f0/262, {exponent}) is {reciprocal:?}"
        );
        assert_ne!(
            powered,
            plane_of(&g, "omitted"),
            "pow(f0/262, {exponent}) is not the default b"
        );
    }
}

fn arguments_of(g: &sva_ast::Graph, target: &str) -> (Vec<sva_engine::Arguments>, usize) {
    let asks = vec![sva_engine::Ask {
        node: target.to_string(),
        representation: sva_engine::Representation::Arguments,
    }];
    let held = render(
        g,
        target,
        RenderConfig::seconds(8_000, 0.05).asking(asks),
        &Tier::default(),
    )
    .expect("a render");
    let id = held.id(target).expect("the target");
    match sva_engine::answer(&held, id, sva_engine::Representation::Arguments)
        .expect("an answer")
        .value
    {
        sva_engine::Output::Arguments(found) => (found, held.buffers.len()),
        other => panic!("expected arguments, got {other:?}"),
    }
}

/// Each binding answers the numbers its solver was handed, the defaults it filled in beside
/// the ones written, and which operand each `max` chose, without rendering a sample.
#[test]
fn the_arguments_reading_answers_what_each_binding_handed_its_builtins() {
    let piano = "0.5*chaigne_askenfelt(f0, vel=vel, b=max(1.4e-4, 4.1e-4*pow(f0/262, 1.9)), \
                 hammer_k=2e10*max(1, pow(f0/523, 0.6)))\n";
    let g = graph_of(
        "arguments-reading",
        &[
            ("piano", piano),
            (
                "chord",
                "@piano(t, f0=130.81, vel=3) + @piano(t, f0=1046.5, vel=3)\n",
            ),
        ],
    );
    let (found, buffers) = arguments_of(&g, "chord");
    assert_eq!(buffers, 0, "a structural reading renders nothing");
    for (f0, b_wins, k_wins) in [(130.81f64, 0, 0), (1046.5, 1, 1)] {
        let node = format!("piano(f0={f0}, vel=3)");
        let held = found
            .iter()
            .find(|a| a.node == node)
            .unwrap_or_else(|| panic!("{node} in {found:?}"));
        let [call] = held.calls.as_slice() else {
            panic!("one solver call: {held:?}")
        };
        assert_eq!(call.name, "chaigne_askenfelt");
        assert_eq!(&piano[call.at.start..call.at.end], "chaigne_askenfelt");
        let arg = |name: &str| {
            call.arguments
                .iter()
                .find(|a| a.name == name)
                .unwrap_or_else(|| panic!("{name} in {call:?}"))
        };
        let b = (1.4e-4f64).max(4.1e-4 * libm::pow(f0 / 262.0, 1.9));
        assert_eq!((arg("b").value, arg("b").written), (b, true), "{f0}");
        let k = 2e10 * 1f64.max(libm::pow(f0 / 523.0, 0.6));
        assert_eq!(arg("hammer_k").value, k, "{f0}");
        assert_eq!((arg("f0").value, arg("vel").value), (f0, 3.0));
        let strike = arg("strike_pos");
        assert!(!strike.written, "a default the solver filled in says so");
        assert_eq!(strike.value, 0.125, "the reference set's own strike");

        let chosen: Vec<(&str, usize)> = held
            .chosen
            .iter()
            .map(|c| (&piano[c.at.start..c.at.end], c.chosen))
            .collect();
        assert_eq!(chosen, [("max", b_wins), ("max", k_wins)], "{f0}");
    }
}

/// A named number that folds to none is refused where it is written, never replaced by the
/// builtin's default; only a filter's cutoff, q and gain may move.
#[test]
fn a_named_argument_that_names_no_number_is_refused_not_defaulted() {
    for written in [
        "chaigne_askenfelt(261.63, b=0.0001*sin(2*pi*t))\n",
        "crop(sample(sin(2*pi*220*t)), 0s, 1s, rise=@lfo)\n",
        "sat(sin(2*pi*220*t), drive=1 + t)\n",
    ] {
        let g = graph_of(
            "moving-argument",
            &[("lfo", "0.1*sin(2*pi*3*t)\n"), ("body", written)],
        );
        let Err(EngineError::Refused(d)) = types(&g, "body") else {
            panic!("{written} types, with its argument silently defaulted");
        };
        assert_eq!(d.code, "engine.non_constant_argument", "{written}: {d:?}");
        let span = d.location.span.expect("located at the call");
        assert!(
            written[span.start..span.end]
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_')
        );
    }
    let swept = graph_of(
        "moving-cutoff",
        &[(
            "body",
            "lowpass(sample(sin(2*pi*1000*t)), cutoff=150 + 7800*t, q=0.707)\n",
        )],
    );
    types(&swept, "body").expect("a filter routes a moving cutoff itself");
}

/// The level a peak at a sine's own frequency leaves it at, in dB, over its settled second half.
fn centre_db(name: &str, files: &[(&str, &str)]) -> f64 {
    let g = graph_of(name, files);
    let held = render(
        &g,
        "body",
        RenderConfig::seconds(48_000, 0.1),
        &Tier::default(),
    )
    .unwrap_or_else(|e| panic!("{files:?}: {e}"));
    let root = held.id("body").expect("the root");
    let output = held.output(root).expect("a render");
    let plane = output.plane(0);
    let settled = &plane[2400..4800];
    let power = settled.iter().map(|s| s * s).sum::<f64>() / settled.len() as f64;
    10.0 * libm::log10(2.0 * power)
}

/// A level is the amplitude ratio `db` writes, in a filter's gain as anywhere: `12db` leaves
/// the centre 12 dB up, written in place or held by a ref, never `12db` read again as dB.
#[test]
fn a_gain_written_in_db_is_the_level_it_names() {
    let tone = "sin(2*pi*1000*t)";
    for (gain, want) in [
        ("12db", 12.0),
        ("-40db", -40.0),
        ("4", 20.0 * libm::log10(4f64)),
    ] {
        for body in [
            format!("peaking(sample({tone}), 1000hz, 2, {gain})\n"),
            format!("sample(peaking({tone}, 1000hz, 2, gain={gain}))\n"),
        ] {
            let found = centre_db("db-gain", &[("body", &body)]);
            assert!(
                (found - want).abs() < 0.01,
                "{body}: {found} dB, not {want}"
            );
        }
    }
    let held = format!("peaking(sample({tone}), 1000hz, 2, gain=@air)\n");
    let found = centre_db("db-gain-ref", &[("air", "12db\n"), ("body", &held)]);
    assert!(
        (found - 12.0).abs() < 0.01,
        "a ref holding 12db: {found} dB"
    );
}

/// A level spelled as a bare number in dB, `-6`, is a ratio no level names: refused, never
/// read as a cut or a tilt.
#[test]
fn a_level_at_or_below_zero_is_refused() {
    for written in [
        "peaking(sample(sin(2*pi*1000*t)), 1000hz, 2, -6)\n",
        "lowshelf(sin(2*pi*1000*t), 300hz, 0.7, gain=0)\n",
        "crop(noise(1, period=1, color=-3), 0s, 0.01s)\n",
    ] {
        let g = graph_of("no-level", &[("body", written)]);
        let Err(refused) = render(
            &g,
            "body",
            RenderConfig::seconds(48_000, 0.01),
            &Tier::default(),
        ) else {
            panic!("{written} names no level and renders nothing");
        };
        assert_eq!(
            refused.code(),
            "engine.level_not_a_ratio",
            "{written}: {refused}"
        );
    }
}
