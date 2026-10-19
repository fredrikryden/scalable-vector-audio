// Concern: states what the shared pipeline answers a front end handing it a target and a condition | Non-concern: what a render sounds like | IO: (text) -> a target, a condition, roots

use sva_core::{
    CliError, Edge, Job, Output, PROBE, Representation, execute, prepared, roots_of, target, until,
};
use sva_engine::{Cmp, Term, Tier, Until};

fn composition(files: &[(&str, &str)]) -> sva_ast::Composition {
    files.iter().copied().collect()
}

/// The target's own ref reads the interval; the expression reads it as `t`.
#[test]
fn a_target_splits_into_its_expression_and_the_interval_its_ref_reads() {
    let held = target("@piano([0, 2b], f0=C4, vel=0.5)").expect("an interval");
    assert_eq!(held.expr, "@piano(t, f0=C4, vel=0.5)");
    assert_eq!(held.interval, Some((Edge::Secs(0.0), Edge::Bars(2.0))));
    for (open, start) in [("@a([1s,))", 1.0), ("@a([500ms, inf))", 0.5)] {
        let held = target(open).expect("an open interval");
        assert_eq!(
            held.interval,
            Some((Edge::Secs(start), Edge::Inf)),
            "{open}"
        );
    }
    let held = target("@a([0, 48000sp])").expect("samples");
    assert_eq!(
        held.interval,
        Some((Edge::Secs(0.0), Edge::Samples(48_000.0)))
    );
    assert_eq!(target("@a*0.5").expect("no interval").interval, None);
    let indexed = target("@a[idx(t) - 1]").expect("an index read, no interval");
    assert_eq!(
        (indexed.expr.as_str(), indexed.interval),
        ("@a[idx(t) - 1]", None)
    );
    let held = target("@a([0, 1s], k=@b[3])").expect("an interval beside an index read");
    assert_eq!(held.expr, "@a(t, k=@b[3])");
}

#[test]
fn an_interval_anywhere_but_the_targets_own_ref_or_without_units_refuses() {
    for refused in [
        "@a(t) + @b([0, 1s])",
        "@a([0, 1s]) * 0.5",
        "@a([2, 3s])",
        "@a([inf, 1s])",
        "@a([0, 1db])",
    ] {
        assert!(
            matches!(target(refused), Err(CliError::Usage { .. })),
            "`{refused}` refuses"
        );
    }
}

#[test]
fn a_condition_reads_comparisons_over_time_and_level() {
    let joined = until(
        "t > 2s or envelope(t) < -60db and t >= 1b",
        44_100,
        Some(2.0),
    )
    .expect("a joined condition");
    let Until::Any(left, right) = joined else {
        panic!("`or` binds loosest");
    };
    assert_eq!(*left, Until::Holds(Term::Time, Cmp::Gt, Term::Number(2.0)));
    let Until::All(level, bars) = *right else {
        panic!("`and` binds tighter");
    };
    assert_eq!(
        *level,
        Until::Holds(Term::Envelope, Cmp::Lt, Term::Number(0.001))
    );
    assert_eq!(*bars, Until::Holds(Term::Time, Cmp::Ge, Term::Number(2.0)));
    for refused in [
        "t > envelope(t)",
        "max(envelope([t, inf))) < -96db",
        "envelope(t - 50ms) < -60db",
        "envelope([t, inf)) < 0.1",
        "t",
        "t > 1s and",
    ] {
        assert!(
            matches!(until(refused, 44_100, None), Err(CliError::Usage { .. })),
            "`{refused}` refuses"
        );
    }
    assert!(matches!(
        until("t > 2b", 44_100, None),
        Err(CliError::BadTempo(_))
    ));
}

/// A level compares with a level in `db`: a bare number beside one has no unit to be read
/// in, and the refusal points at it.
#[test]
fn a_level_compared_with_a_bare_number_refuses_where_it_is_written() {
    for (text, at) in [
        ("envelope(t) < -60", "`-60` (at bytes 14..17)"),
        ("t > 1s or 0.001 > envelope(t)", "`0.001` (at bytes 10..15)"),
    ] {
        let Err(CliError::Usage { message: why, .. }) = until(text, 44_100, None) else {
            panic!("`{text}` compares a level with no unit");
        };
        assert!(why.contains(at), "{why}");
    }
}

fn asking<'a>(held: &'a sva_ast::Composition, target: &'a str) -> Job<'a> {
    Job::over(held, target)
}

#[test]
fn an_interval_that_holds_no_sample_or_names_bars_with_no_tempo_refuses() {
    let held = composition(&[("tone", "crop(sin(2*pi*220*t), 0s, 1s)\n")]);
    assert!(execute(asking(&held, "@tone([0, 0.5s])"), &Tier::default()).is_ok());
    assert!(matches!(
        execute(asking(&held, "@tone([1s, 0.5s])"), &Tier::default()),
        Err(CliError::Usage { .. })
    ));
    assert!(matches!(
        execute(asking(&held, "@tone([0, 1b])"), &Tier::default()),
        Err(CliError::BadTempo(_))
    ));
}

/// What an expression reads are the roots a load pulls, and every reserved variable besides.
#[test]
fn the_roots_of_a_target_are_what_its_math_reads() {
    let source = composition(&[("master", "@kick\n"), ("kick", "sin(2*pi*60*t)\n")]);
    let holds = |target, name: &str| {
        roots_of(&source, target)
            .unwrap()
            .contains(&name.to_string())
    };
    assert!(holds("@kick*0.5", "kick"), "argv math pulls what it reads");
    assert!(
        holds("sin(t)", "variables/bpm"),
        "and every reserved variable, which an interval or `--until` may read"
    );
}

/// Math is never looked up as a file, so no length of expression reads as a path too long.
#[test]
fn an_expression_of_any_length_is_math_and_never_a_path() {
    let dir = std::env::temp_dir().join(format!("sva-core-long-expr-{:x}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let long = (0..20)
        .map(|k| {
            format!(
                "crop(saw(2*pi*55*(t-{}s)),{}s,{}s)",
                4 * k,
                4 * k,
                4 * k + 1
            )
        })
        .collect::<Vec<_>>()
        .join(" + ");
    let path_shaped = vec!["t"; 200].join("-");
    let roots: Vec<_> = [long, path_shaped]
        .iter()
        .inspect(|long| assert!(long.len() > 255, "longer than any file name"))
        .map(|long| roots_of(&sva_ast::Dir::at(&dir), long))
        .collect();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(roots.iter().all(Result::is_ok), "{roots:?}");
}

/// `bpm` and `meter` come as a pair: only a pair turns a bar into seconds.
#[test]
fn a_bar_span_is_settled_by_a_whole_tempo_pair_and_by_nothing_less() {
    let whole = composition(&[
        ("variables/bpm", "120\n"),
        ("variables/meter", "4/4\n"),
        ("hit-1b", "sin(2*pi*220*t)\n"),
        ("master", "@hit-1b\n"),
    ]);
    let graph = prepared(&whole).expect("a tempo settles the span");
    assert_eq!(
        graph.tempo().map(|tempo| tempo.seconds_per_bar()),
        Some(2.0),
        "four beats at 120 bpm"
    );

    for half in [
        composition(&[
            ("variables/bpm", "120\n"),
            ("hit-1b", "sin(t)\n"),
            ("master", "@hit-1b\n"),
        ]),
        composition(&[
            ("variables/meter", "4/4\n"),
            ("hit-1b", "sin(t)\n"),
            ("master", "@hit-1b\n"),
        ]),
        composition(&[
            ("variables/bpm", "0\n"),
            ("variables/meter", "4/4\n"),
            ("hit-1b", "sin(t)\n"),
            ("master", "@hit-1b\n"),
        ]),
    ] {
        let Err(CliError::Refusals(refused)) = prepared(&half) else {
            panic!("half a tempo settles nothing");
        };
        assert_eq!(refused[0].code, sva_ast::DiagCode::BadTempo);
    }
}

/// A refusal names the target as the caller wrote it, never the node the pipeline defined it as.
#[test]
fn a_refusal_names_the_target_as_written() {
    let held = composition(&[("tone", "sin(2*pi*220*t)\n")]);
    let Err(refused) = execute(asking(&held, "@tone"), &Tier::default()) else {
        panic!("nothing ends an endless tone");
    };
    let message = refused.message();
    assert!(message.contains("(@tone)"), "{message}");
    assert!(!message.contains("probe"), "{message}");
}

/// A target that parses answers off the expression it is: what it leaves unbound is the
/// engine's own refusal, what it fails to parse the parser's.
#[test]
fn a_target_answers_off_the_expression_it_is() {
    let held = composition(&[("master", "crop(sin(2*pi*220*t), 0s, 1s)\n")]);
    for (text, code) in [
        ("2^3", "validation_error"),
        ("sin(q*t)", "validation_error"),
        ("sum(k, 1, 3, k*", "validation_error"),
    ] {
        let Err(refused) = execute(asking(&held, text), &Tier::default()) else {
            panic!("`{text}` renders nothing");
        };
        assert_eq!(refused.code(), code, "{text}: {refused:?}");
    }
    let Err(refused) = execute(asking(&held, "sin(q*t)"), &Tier::default()) else {
        panic!("`q` is bound nowhere, so nothing renders");
    };
    assert!(
        refused.message().contains('q'),
        "the engine's own message names the free variable: {}",
        refused.message()
    );
    assert!(execute(asking(&held, "@master"), &Tier::default()).is_ok());
}

/// A level is every channel's: a tone in the second channel beside silence in the first is
/// half its power, and a quiet `until` does not hold over it.
#[test]
fn a_level_reads_every_channel() {
    let held = composition(&[]);
    let job = Job {
        until: Some("envelope(t) < -60db"),
        ..asking(&held, "join(0, crop(sample(sin(2*pi*441*t)), 0s, 0.5s))")
    };
    let rendered = execute(job, &Tier::default()).expect("a stereo tone");
    let envelope = Representation::Envelope {
        frame_secs: Some(0.1),
    };
    let Output::Envelope(frames) = rendered.answer(PROBE, envelope).expect("frames").value else {
        panic!("a measured envelope answers in frames");
    };
    assert_eq!(frames.len(), 5, "the render ran its whole half second");
    for frame in frames {
        assert!((frame.rms - 0.5).abs() < 0.01, "sqrt(1/2 / 2): {frame:?}");
        assert!(frame.peak > 0.99, "{frame:?}");
    }
}

/// A reading's time arguments are times as a score writes them: a bar is what the render's
/// tempo says, the same at any argument, and with no tempo to count it in a bar refuses.
#[test]
fn a_readings_times_count_bars_in_the_renders_tempo() {
    let reading = |text: &str| sva_core::asked(&sva_core::call(text).unwrap()).unwrap();
    let framed = |text: &str, per_bar| reading(text).representation(44_100, per_bar);
    assert_eq!(
        framed("pitch(frame=1b/8)", Some(2.0)).unwrap(),
        framed("pitch(frame=250ms)", None).unwrap(),
        "an eighth of a two-second bar"
    );
    assert_eq!(
        framed(
            "self_similarity(cell=1b/2, smooth=2b, window=4b)",
            Some(2.0)
        )
        .unwrap(),
        framed("self_similarity(cell=1s, smooth=4s, window=8s)", None).unwrap(),
        "every time argument counts bars alike"
    );
    let Err(CliError::BadTempo(why)) = framed("pitch(frame=1b/8)", None) else {
        panic!("a bar with no tempo has no length");
    };
    assert!(why.contains("pitch(frame=1b/8)"), "{why}");
    let Err(CliError::Usage { code, .. }) =
        sva_core::asked(&sva_core::call("pitch(frame=1/8b)").unwrap())
    else {
        panic!("one over eight bars is a rate, not a time");
    };
    assert_eq!(code, "query.bad_value");

    let held = composition(&[
        ("variables/bpm", "120\n"),
        ("variables/meter", "4/4\n"),
        ("tone", "crop(sin(2*pi*220*t), 0s, 1s)\n"),
    ]);
    let asked = [reading("pitch(frame=1b/8)")];
    let job = Job {
        asked: &asked,
        ..Job::over(&held, "sample(@tone([0, 1s]))")
    };
    let rendered = execute(job, &Tier::default()).expect("a render");
    let Output::Pitch(frames) = rendered.answered(&asked[0]).expect("a pitch").value else {
        panic!("a pitch reading");
    };
    let steps: Vec<f64> = frames.iter().map(|f| f.t_secs).collect();
    assert_eq!(
        steps,
        [0.0, 0.25, 0.5, 0.75],
        "a frame every eighth of a bar"
    );
}

/// A time argument counts samples at the render's rate: a frame rounding to none, or a time
/// past the 2^53 a count holds exactly, refuses naming the least or most it may be.
#[test]
fn a_time_no_whole_sample_count_holds_refuses_at_the_rate() {
    let reading = |text: &str| sva_core::asked(&sva_core::call(text).unwrap()).unwrap();
    for (text, rate, says) in [
        (
            "envelope(frame=0.00001s)",
            44_100,
            "at least 0.000011337868480725624s",
        ),
        (
            "self_similarity(cell=1e15)",
            44_100,
            "at most 204244881059.88644s",
        ),
        (
            "self_similarity(smooth=1e300)",
            44_100,
            "at most 204244881059.88644s",
        ),
        (
            "periodicity(window=1e15)",
            44_100,
            "at most 204244881059.88644s",
        ),
        (
            "envelope(frame=1e15)",
            44_100,
            "at most 204244881059.88644s",
        ),
    ] {
        let Err(CliError::Usage { code, message }) = reading(text).representation(rate, None)
        else {
            panic!("{text} counts no whole samples at {rate} Hz");
        };
        assert_eq!(code, "query.bad_value", "{text}");
        assert!(message.contains(says), "{says}: {message}");
    }
    for (text, rate) in [
        ("envelope(frame=0.00001s)", 96_000),
        ("self_similarity(cell=1e11)", 44_100),
    ] {
        assert!(reading(text).representation(rate, None).is_ok(), "{text}");
    }
}
