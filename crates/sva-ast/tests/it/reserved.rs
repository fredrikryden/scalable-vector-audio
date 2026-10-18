// Concern: proves each reserved variable is found one way, lowered where read, and again where its value changes | Non-concern: score line checks (score.rs) | IO: (a composition, an edit) -> nodes
use sva_ast::{Composition, DiagCode, Expr, FileSpan, SpanUnit, load, load_beside, parse_expr};

fn tempo(bpm: &str) -> Composition {
    Composition::from_iter([
        ("variables/bpm", bpm),
        ("variables/meter", "4/4\n"),
        ("kick", "sin(2*pi*50*t)\n"),
        ("swung", "@kick(t - 0.02b)*0.9\n"),
        ("tiled", "crop(@kick(t % 1b), 0s, 16b)\n"),
        (
            "phrase-1b",
            "{\"at\": \"0b\", \"node\": \"@kick\"}\n{\"at\": \"0.01b\", \"node\": \"@kick\"}\n",
        ),
        ("tail-1.5s", "@kick\n"),
    ])
}

fn expr(text: &str) -> Expr {
    parse_expr(text).expect("an expression")
}

/// A literal, a score's time and a file's span are one read, so the three always agree.
#[test]
fn a_bar_is_beats_times_sixty_over_the_bpm_wherever_it_is_written() {
    let mut g = load(&tempo("120\n")).expect("a composition");
    g.lower_reserved(None).expect("a tempo");
    assert_eq!(
        g.expr("swung"),
        Some(&expr("@kick(t - 0.02*(240/120))*0.9"))
    );
    assert_eq!(
        g.expr("tiled"),
        Some(&expr("crop(@kick(t % (1*(240/120))), 0s, 16*(240/120))"))
    );
    assert_eq!(
        g.expr("phrase-1b"),
        Some(&expr("@kick + @kick(t - 0.01*(240/120))"))
    );
    let seconds = |amount| {
        Some(FileSpan {
            amount,
            unit: SpanUnit::Seconds,
        })
    };
    assert_eq!(g.span("phrase-1b"), seconds(2.0));
    assert_eq!(g.span("tail-1.5s"), seconds(1.5));
    assert_eq!(g.tempo().map(|t| t.seconds_per_bar()), Some(2.0));
}

#[test]
fn a_bar_with_no_tempo_refuses_naming_its_node() {
    let mut g = load(&Composition::from_iter([
        ("kick", "sin(2*pi*50*t)\n"),
        ("swung", "@kick(t - 0.02b)\n"),
    ]))
    .expect("a composition");
    let refused = g.lower_reserved(None).unwrap_err();
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert_eq!(refused[0].code, DiagCode::BadTempo);
    assert_eq!(refused[0].at.path, "swung");
}

/// `variables/` is the one place a reserved variable is read from: a `bpm` at the root is a
/// node like any other.
#[test]
fn only_a_node_under_variables_is_a_reserved_variable() {
    let mut g = load(&Composition::from_iter([
        ("bpm", "120\n"),
        ("meter", "4/4\n"),
        ("swung", "sin(2*pi*50*(t - 1b))\n"),
    ]))
    .expect("a composition");
    assert_eq!(g.tempo(), None);
    assert_eq!(
        g.lower_reserved(None).unwrap_err()[0].code,
        DiagCode::BadTempo
    );
}

#[test]
fn a_tempo_that_is_no_tempo_refuses_at_its_variable() {
    for (bpm, meter, at) in [
        ("0\n", "4/4\n", "variables/bpm"),
        ("120\n", "four/4\n", "variables/meter"),
    ] {
        let mut source = tempo(bpm);
        source.insert("variables/meter", meter);
        let refused = load(&source).unwrap().lower_reserved(None).unwrap_err();
        assert_eq!(refused[0].code, DiagCode::BadTempo, "{refused:?}");
        assert_eq!(refused[0].at.path, at);
    }
    let mut g = load(&Composition::from_iter([("variables/bpm", "120\n")])).unwrap();
    assert_eq!(
        g.lower_reserved(None).unwrap_err()[0].code,
        DiagCode::BadTempo
    );
}

#[test]
fn a_scale_names_a_scale_file_by_a_bare_ref_or_refuses_at_its_variable() {
    for scale in ["3\n", "@../voice\n", "@../voice(t - 1s)\n"] {
        let mut g = load(&Composition::from_iter([
            ("variables/scale", scale),
            ("voice", "sin(2*pi*440*t)\n"),
        ]))
        .unwrap_or_else(|r| panic!("{scale}: {r:?}"));
        let refused = g.lower_reserved(None).unwrap_err();
        assert_eq!(refused[0].code, DiagCode::BadScale, "{scale}");
        assert_eq!(refused[0].at.path, "variables/scale");
    }
}

#[test]
fn a_degree_is_the_tonics_ref_times_its_ratio_and_needs_a_tonic() {
    let source = Composition::from_iter([
        ("variables/key", "A3\n"),
        ("variables/scale", "@../scales/major\n"),
        ("scales/major", "200\n400\n500\n700\n900\n1100\n1200\n"),
        ("lead/one", "sin(2*pi*deg(8)*t)\n"),
    ]);
    let mut g = load(&source).unwrap();
    g.lower_reserved(None).expect("a key and a scale");
    assert_eq!(
        g.expr("lead/one"),
        Some(&expr("sin(2*pi*(@../variables/key*2)*t)"))
    );
    let mut untoned = load(&Composition::from_iter([
        ("variables/scale", "@../scales/major\n"),
        ("scales/major", "200\n400\n500\n700\n900\n1100\n1200\n"),
        ("lead/one", "sin(2*pi*deg(8)*t)\n"),
    ]))
    .unwrap();
    let refused = untoned.lower_reserved(None).unwrap_err();
    assert_eq!(refused[0].at.path, "lead/one");
    assert!(refused[0].reason.contains("variables/key"), "{refused:?}");
}

/// A held node is lowered again where an edit changes a value it read, and only there.
#[test]
fn an_edit_to_the_tempo_lowers_every_held_node_counting_in_bars_again() {
    let mut source = tempo("120\n");
    let mut held = load(&source).unwrap();
    held.lower_reserved(None).unwrap();

    source.insert("variables/bpm", "240\n");
    let mut delta = load_beside(&source, &["swung", "tail-1.5s"], &held).unwrap();
    delta.lower_reserved(Some(&held)).unwrap();
    assert_eq!(
        delta.expr("swung"),
        Some(&expr("@kick(t - 0.02*(240/240))*0.9"))
    );
    assert!(
        !delta.defines("tail-1.5s") && !delta.defines("kick"),
        "a node no bar reads is held as it was"
    );
}
